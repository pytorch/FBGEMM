/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */
#pragma once

#include <c10/macros/Macros.h>
#include <c10/util/ArrayRef.h>
#include <c10/util/Deprecated.h>
#include <c10/util/Exception.h>
#include <c10/util/irange.h>
#include <stdint.h>
#include <cstddef>

namespace fbgemm_gpu {

// The PtrTraits argument to the TensorAccessor/GenericPackedTensorAccessor
// is used to enable the __restrict__ keyword/modifier for the data
// passed to cuda.
template <typename T>
struct DefaultPtrTraits {
  typedef T* PtrType;
};

#if defined(__CUDACC__) || defined(__HIPCC__)
template <typename T>
struct RestrictPtrTraits {
  typedef T* __restrict__ PtrType;
};
#endif

// TensorAccessorBase and TensorAccessor are used for both CPU and CUDA tensors.
// For CUDA tensors it is used in device code (only). This means that we
// restrict ourselves to functions and types available there (e.g.
// at::IntArrayRef isn't).

// The PtrTraits argument is only relevant to cuda to support `__restrict__`
// pointers.
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class TensorAccessorBase {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : data_(data_),
        sizes_(sizes_),
        strides_(strides_),
        ptr_name_(ptr_name_),
        func_name_(func_name_) {
    this->numel_ = 0;
    for (size_t d = 0; d < N; d++) {
      this->numel_ += sizes_[d];
    }
  }
  C10_HOST at::IntArrayRef sizes() const {
    return at::IntArrayRef(sizes_, N);
  }
  C10_HOST at::IntArrayRef strides() const {
    return at::IntArrayRef(strides_, N);
  }
  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }
  C10_HOST_DEVICE void check_access(index_t idx) const {
    if (idx < 0) {
      printf(
          "ERROR: idx < 0, tensor %s in %s, idx %lld\n",
          this->ptr_name_,
          this->func_name_,
          static_cast<int64_t>(idx));
    } else if (idx >= this->numel_) {
      printf(
          "ERROR: idx >= numel, tensor %s in %s, idx %lld, numel %lld\n",
          this->ptr_name_,
          this->func_name_,
          static_cast<int64_t>(idx),
          static_cast<int64_t>(this->numel_));
    }
    CUDA_KERNEL_ASSERT(idx >= 0)
    CUDA_KERNEL_ASSERT(idx < this->numel_);
  }

 protected:
  PtrType data_;
  const index_t* sizes_;
  const index_t* strides_;
  index_t numel_;
  const char* ptr_name_;
  const char* func_name_;
};

// The `TensorAccessor` is typically instantiated for CPU `Tensor`s using
// `Tensor.accessor<T, N>()`.
// For CUDA `Tensor`s, `GenericPackedTensorAccessor` is used on the host and
// only indexing on the device uses `TensorAccessor`s.
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class TensorAccessor : public TensorAccessorBase<T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : TensorAccessorBase<T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_,
            ptr_name_,
            func_name_) {}

  C10_HOST_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](
      index_t i) {
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1,
        this->ptr_name_,
        this->func_name);
  }

  C10_HOST_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](
      index_t i) const {
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        this->sizes_ + 1,
        this->strides_ + 1,
        this->ptr_name_,
        this->func_name);
  }
};

template <typename T, template <typename U> class PtrTraits, typename index_t>
class TensorAccessor<T, 1, PtrTraits, index_t>
    : public TensorAccessorBase<T, 1, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST_DEVICE TensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : TensorAccessorBase<T, 1, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_,
            ptr_name_,
            func_name_) {}
  C10_HOST_DEVICE T& operator[](index_t i) {
    this->check_access(this->strides_[0] * i);
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->data_[this->strides_[0] * i];
  }
  C10_HOST_DEVICE const T& operator[](index_t i) const {
    this->check_access(this->strides_[0] * i);
    // NOLINTNEXTLINE(clang-analyzer-core.NullDereference)
    return this->data_[this->strides_[0] * i];
  }
};

// GenericPackedTensorAccessorBase and GenericPackedTensorAccessor are used on
// for CUDA `Tensor`s on the host and as In contrast to `TensorAccessor`s, they
// copy the strides and sizes on instantiation (on the host) in order to
// transfer them on the device when calling kernels. On the device, indexing of
// multidimensional tensors gives to `TensorAccessor`s. Use RestrictPtrTraits as
// PtrTraits if you want the tensor's data pointer to be marked as __restrict__.
// Instantiation from data, sizes, strides is only needed on the host and
// std::copy isn't available on the device, so those functions are host only.
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class GenericPackedTensorAccessorBase {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST GenericPackedTensorAccessorBase(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : data_(data_) {
    std::copy(sizes_, sizes_ + N, std::begin(this->sizes_));
    std::copy(strides_, strides_ + N, std::begin(this->strides_));
    // Compute numel_
    this->numel_ = 0;
    for (size_t d = 0; d < N; d++) {
      this->numel_ += sizes_[d];
    }
    auto ptr_name_len = std::min(strlen(ptr_name_), static_cast<size_t>(16));
    std::memcpy(this->ptr_name_, ptr_name_, sizeof(const char) * ptr_name_len);
    auto func_name_len = std::min(strlen(func_name_), static_cast<size_t>(64));
    std::memcpy(
        this->func_name_, func_name_, sizeof(const char) * func_name_len);
  }

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <
      typename source_index_t,
      class = typename std::enable_if<
          std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST GenericPackedTensorAccessorBase(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : data_(data_) {
    for (const auto i : c10::irange(N)) {
      this->sizes_[i] = sizes_[i];
      this->strides_[i] = strides_[i];
    }
    // Compute numel_
    this->numel_ = 0;
    for (size_t d = 0; d < N; d++) {
      this->numel_ += sizes_[d];
    }
    auto ptr_name_len = std::min(strlen(ptr_name_), static_cast<size_t>(16));
    std::memcpy(this->ptr_name_, ptr_name_, sizeof(const char) * ptr_name_len);
    auto func_name_len = std::min(strlen(func_name_), static_cast<size_t>(64));
    std::memcpy(
        this->func_name_, func_name_, sizeof(const char) * func_name_len);
  }

  C10_HOST_DEVICE void check_access(index_t idx) const {
    if (idx < 0) {
      printf(
          "ERROR: idx < 0, tensor %s in %s, idx %lld\n",
          this->ptr_name_,
          this->func_name_,
          static_cast<int64_t>(idx));
    } else if (idx >= this->numel_) {
      printf(
          "ERROR: idx >= numel, tensor %s in %s, idx %lld, numel %lld\n",
          this->ptr_name_,
          this->func_name_,
          static_cast<int64_t>(idx),
          static_cast<int64_t>(this->numel_));
    }
    CUDA_KERNEL_ASSERT(idx >= 0)
    CUDA_KERNEL_ASSERT(idx < this->numel_)
  }

  C10_HOST_DEVICE index_t stride(index_t i) const {
    return strides_[i];
  }
  C10_HOST_DEVICE index_t size(index_t i) const {
    return sizes_[i];
  }
  C10_HOST_DEVICE PtrType data() {
    return data_;
  }
  C10_HOST_DEVICE const PtrType data() const {
    return data_;
  }

 protected:
  PtrType data_;
  index_t sizes_[N];
  index_t strides_[N];
  index_t numel_;
  char ptr_name_[16];
  char func_name_[64];
  C10_HOST void bounds_check_(index_t i) const {
    TORCH_CHECK_INDEX(
        0 <= i && i < index_t{N},
        "Index ",
        i,
        " is not within bounds of a tensor of dimension ",
        N);
  }
};

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
class GenericPackedTensorAccessor
    : public GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;

  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_,
            ptr_name_,
            func_name_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <
      typename source_index_t,
      class = typename std::enable_if<
          std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : GenericPackedTensorAccessorBase<T, N, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_,
            ptr_name_,
            func_name_) {}

  C10_DEVICE TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](
      index_t i) {
    index_t* new_sizes = this->sizes_ + 1;
    index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        new_sizes,
        new_strides,
        this->ptr_name_,
        this->func_name_);
  }

  C10_DEVICE const TensorAccessor<T, N - 1, PtrTraits, index_t> operator[](
      index_t i) const {
    const index_t* new_sizes = this->sizes_ + 1;
    const index_t* new_strides = this->strides_ + 1;
    return TensorAccessor<T, N - 1, PtrTraits, index_t>(
        this->data_ + this->strides_[0] * i,
        new_sizes,
        new_strides,
        this->ptr_name_,
        this->func_name_);
  }

  /// Returns a PackedTensorAccessor of the same dimension after transposing the
  /// two dimensions given. Does not actually move elements; transposition is
  /// made by permuting the size/stride arrays. If the dimensions are not valid,
  /// asserts.
  C10_HOST GenericPackedTensorAccessor<T, N, PtrTraits, index_t> transpose(
      index_t dim1,
      index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    GenericPackedTensorAccessor<T, N, PtrTraits, index_t> result(
        this->data_, this->sizes_, this->strides_);
    std::swap(result.strides_[dim1], result.strides_[dim2]);
    std::swap(result.sizes_[dim1], result.sizes_[dim2]);
    return result;
  }
};

template <typename T, template <typename U> class PtrTraits, typename index_t>
class GenericPackedTensorAccessor<T, 1, PtrTraits, index_t>
    : public GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t> {
 public:
  typedef typename PtrTraits<T>::PtrType PtrType;
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const index_t* sizes_,
      const index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_,
            ptr_name_,
            func_name_) {}

  // if index_t is not int64_t, we want to have an int64_t constructor
  template <
      typename source_index_t,
      class = typename std::enable_if<
          std::is_same<source_index_t, int64_t>::value>::type>
  C10_HOST GenericPackedTensorAccessor(
      PtrType data_,
      const source_index_t* sizes_,
      const source_index_t* strides_,
      const char* ptr_name_,
      const char* func_name_)
      : GenericPackedTensorAccessorBase<T, 1, PtrTraits, index_t>(
            data_,
            sizes_,
            strides_,
            ptr_name_,
            func_name_) {}

  C10_DEVICE T& operator[](index_t i) {
    this->check_access(this->strides_[0] * i);
    return this->data_[this->strides_[0] * i];
  }
  C10_DEVICE const T& operator[](index_t i) const {
    this->check_access(this->strides_[0] * i);
    return this->data_[this->strides_[0] * i];
  }

  // Same as in the general N-dimensional case, but note that in the
  // 1-dimensional case the returned PackedTensorAccessor will always be an
  // identical copy of the original
  C10_HOST GenericPackedTensorAccessor<T, 1, PtrTraits, index_t> transpose(
      index_t dim1,
      index_t dim2) const {
    this->bounds_check_(dim1);
    this->bounds_check_(dim2);
    return GenericPackedTensorAccessor<T, 1, PtrTraits, index_t>(
        this->data_, this->sizes_, this->strides_);
  }
};

// Can't put this directly into the macro function args because of commas
#define AT_X GenericPackedTensorAccessor<T, N, PtrTraits, index_t>

// Old name for `GenericPackedTensorAccessor`
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits,
    typename index_t = int64_t>
C10_DEFINE_DEPRECATED_USING(PackedTensorAccessor, AT_X)

#undef AT_X

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor32 =
    GenericPackedTensorAccessor<T, N, PtrTraits, int32_t>;

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = DefaultPtrTraits>
using PackedTensorAccessor64 =
    GenericPackedTensorAccessor<T, N, PtrTraits, int64_t>;

} // namespace fbgemm_gpu

#ifdef FBGEMM_GPU_MEMCHECK
namespace pta = fbgemm_gpu;
#else
namespace pta = at;
#endif

#ifdef FBGEMM_GPU_MEMCHECK
template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = at::DefaultPtrTraits,
    typename index_t = int64_t>
const fbgemm_gpu::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>
make_generic_packed_tensor_accessor(
    at::Tensor& tensor,
    const char* ptr_name,
    const char* func_name) {
  static_assert(
      N > 0,
      "accessor is used for indexing tensor, for scalars use *data_ptr<T>()");
  TORCH_CHECK(
      tensor.dim() == N,
      "TensorAccessor expected ",
      N,
      " dims but tensor has ",
      tensor.dim());
  return fbgemm_gpu::GenericPackedTensorAccessor<T, N, PtrTraits, index_t>(
      static_cast<typename PtrTraits<T>::PtrType>(tensor.data_ptr<T>()),
      tensor.sizes().data(),
      tensor.strides().data(),
      ptr_name,
      func_name);
}
#endif

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
const pta::PackedTensorAccessor32<T, N, PtrTraits>
make_packed_tensor_accessor32(
#ifdef FBGEMM_GPU_MEMCHECK
    at::Tensor& tensor,
    const char* ptr_name,
    const char* func_name) {
#else
    at::Tensor& tensor) {
#endif
  TORCH_CHECK(
      tensor.numel() <=
          static_cast<int64_t>(std::numeric_limits<int32_t>::max()),
      "numel needs to be smaller than int32_t max; otherwise, please use packed_accessor64");
#ifdef FBGEMM_GPU_MEMCHECK
  return make_generic_packed_tensor_accessor<T, N, PtrTraits, int32_t>(
      tensor, ptr_name, func_name);
#else
  return tensor.packed_accessor32<T, N, PtrTraits>();
#endif
}

template <
    typename T,
    size_t N,
    template <typename U> class PtrTraits = at::DefaultPtrTraits>
const pta::PackedTensorAccessor64<T, N, PtrTraits>
make_packed_tensor_accessor64(
#ifdef FBGEMM_GPU_MEMCHECK
    at::Tensor& tensor,
    const char* ptr_name,
    const char* func_name) {
  return make_generic_packed_tensor_accessor<T, N, PtrTraits, int64_t>(
      tensor, ptr_name, func_name);
#else
    at::Tensor& tensor) {
  return tensor.packed_accessor64<T, N, PtrTraits>();
#endif
}

#ifdef FBGEMM_GPU_MEMCHECK
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
