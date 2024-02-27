/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <ATen/ATen.h>
#include <cstdint>
#include <optional>
#include <string>

inline bool torch_tensor_on_cpu_check(const at::Tensor& ten) {
  return ten.is_cpu();
}

inline bool torch_tensor_on_cpu_check(const c10::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_cpu_check(ten.value());
}

inline std::optional<int64_t> get_device_index_from_tensor(
    const at::Tensor& ten) {
  return {ten.device().index()};
}

inline std::optional<int64_t> get_device_index_from_tensor(
    const c10::optional<at::Tensor>& ten) {
  if (ten) {
    return {ten->device().index()};
  } else {
    return {};
  }
}

inline std::string torch_tensor_device_name(const at::Tensor& ten) {
  return c10::DeviceTypeName(ten.device().type());
}

inline std::string torch_tensor_device_name(
    const c10::optional<at::Tensor>& ten) {
  if (ten.has_value()) {
    return torch_tensor_device_name(ten.value());
  } else {
    return "N/A";
  }
}

inline bool torch_tensor_on_same_device_check(
    const at::Tensor& ten1,
    const at::Tensor& ten2) {
  return ten1.get_device() == ten2.get_device();
}

inline bool torch_tensor_on_same_device_check(
    const at::Tensor& ten1,
    const c10::optional<at::Tensor>& ten2) {
  return !ten2.has_value() || ten1.get_device() == ten2->get_device();
}

inline bool torch_tensor_undefined(const at::Tensor& ten) {
  return ten.defined();
}

inline bool torch_tensor_undefined(const c10::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_undefined(ten.value());
}

inline bool torch_tensor_on_cuda_gpu_check(const at::Tensor& ten) {
  return ten.is_cuda();
}

inline bool torch_tensor_on_cuda_gpu_check(
    const c10::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_cuda_gpu_check(ten.value());
}

inline bool torch_tensor_empty_or_on_cuda_gpu_check(const at::Tensor& ten) {
  return (ten.numel() == 0) || ten.is_cuda();
}

inline bool torch_tensor_empty_or_on_cuda_gpu_check(
    const c10::optional<at::Tensor>& ten) {
  return !ten.has_value() ||
      torch_tensor_empty_or_on_cuda_gpu_check(ten.value());
}

inline bool torch_tensor_empty_or_on_cpu_check(const at::Tensor& ten) {
  return (ten.numel() == 0) || ten.is_cpu();
}

inline bool torch_tensor_empty_or_on_cpu_check(
    const c10::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_empty_or_on_cpu_check(ten.value());
}

#define DISPATCH_TO_CUDA(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::CUDA, TORCH_FN(function)))

#define DISPATCH_TO_CPU(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::CPU, TORCH_FN(function)))

#define DISPATCH_TO_META(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::Meta, TORCH_FN(function)))

#define DISPATCH_TO_ALL(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::CatchAll, TORCH_FN(function)))

#define DISPATCH_TO_AUTOGRAD(name, function) \
  m.impl(name, torch::dispatch(c10::DispatchKey::Autograd, TORCH_FN(function)))

#define DISPATCH_TO_AUTOGRAD_CUDA(name, function) \
  m.impl(                                         \
      name,                                       \
      torch::dispatch(c10::DispatchKey::AutogradCUDA, TORCH_FN(function)))

#define DISPATCH_TO_AUTOGRAD_META(name, function) \
  m.impl(                                         \
      name,                                       \
      torch::dispatch(c10::DispatchKey::AutogradMETA, TORCH_FN(function)))

#define TENSOR_ON_CPU(x)                                      \
  TORCH_CHECK(                                                \
      torch_tensor_on_cpu_check(x),                           \
      #x " must be a CPU tensor; it is currently on device ", \
      torch_tensor_device_name(x))

#define TENSOR_EMPTY_OR_ON_CPU(x)                                      \
  TORCH_CHECK(                                                         \
      torch_tensor_empty_or_on_cpu_check(x),                           \
      #x " must be empty or a CPU tensor; it is currently on device ", \
      torch_tensor_device_name(x))

#define TENSOR_ON_CUDA_GPU(x)                                  \
  TORCH_CHECK(                                                 \
      torch_tensor_on_cuda_gpu_check(x),                       \
      #x " must be a CUDA tensor; it is currently on device ", \
      torch_tensor_device_name(x))

#define TENSOR_EMPTY_OR_ON_CUDA_GPU(x)                                  \
  TORCH_CHECK(                                                          \
      torch_tensor_empty_or_on_cuda_gpu_check(x),                       \
      #x " must be empty or a CUDA tensor; it is currently on device ", \
      torch_tensor_device_name(x))

#define TENSORS_EMPTY_OR_ON_SAME_DEVICE(x, y)                           \
  TORCH_CHECK(                                                          \
      torch_tensor_on_same_device_check(x, y) || (x.numel() == 0),      \
      #x " must be empty or a CUDA tensor; it is currently on device ", \
      torch_tensor_device_name(x))

#define TENSORS_ON_SAME_DEVICE(x, y)                                       \
  TORCH_CHECK(                                                             \
      torch_tensor_on_same_device_check(x, y),                             \
      #x " must be on the same device as " #y "! " #x " is currently on ", \
      torch_tensor_device_name(x),                                         \
      #y " is currently on ",                                              \
      torch_tensor_device_name(y))

#define TENSORS_HAVE_SAME_TYPE(x, y)                       \
  TORCH_CHECK(                                             \
      (x).dtype() == (y).dtype(),                          \
      #x " must have the same type as " #y " types were ", \
      (x).dtype().name(),                                  \
      " and ",                                             \
      (y).dtype().name())

#define TENSOR_NDIM_EQUALS(ten, dims)      \
  TORCH_CHECK(                             \
      (ten).ndimension() == (dims),        \
      "Tensor '" #ten "' must have " #dims \
      " dimension(s). "                    \
      "Found ",                            \
      (ten).ndimension())

#define TENSOR_CONTIGUOUS(x) \
  TORCH_CHECK((x).is_contiguous(), #x " must be contiguous")

#define TENSOR_CONTIGUOUS_AND_ON_CPU(x) \
  TENSOR_ON_CPU(x);                     \
  TENSOR_CONTIGUOUS(x)

#define TENSOR_CONTIGUOUS_AND_ON_CUDA_GPU(x) \
  TENSOR_ON_CUDA_GPU(x);                     \
  TENSOR_CONTIGUOUS(x)

#define TENSOR_NDIM_IS_GE(ten, dims)         \
  TORCH_CHECK(                               \
      (ten).dim() >= (dims),                 \
      "Tensor '" #ten "' must have >=" #dims \
      " dimension(s). "                      \
      "Found ",                              \
      (ten).ndimension())

#define TENSOR_TYPE_MUST_BE(ten, typ)                                      \
  TORCH_CHECK(                                                             \
      (ten).scalar_type() == typ,                                          \
      "Tensor '" #ten "' must have scalar type " #typ " but it had type ", \
      (ten).dtype().name())

#define TENSOR_NDIM_EXCEEDS(ten, dims)               \
  TORCH_CHECK(                                       \
      (ten).dim() > (dims),                          \
      "Tensor '" #ten "' must have more than " #dims \
      " dimension(s). "                              \
      "Found ",                                      \
      (ten).ndimension())

#define TENSORS_HAVE_SAME_NUMEL(x, y)                                  \
  TORCH_CHECK(                                                         \
      (x).numel() == (y).numel(),                                      \
      #x " must have the same number of elements as " #y " They had ", \
      (x).numel(),                                                     \
      " and ",                                                         \
      (y).numel())

template <typename... Tensors>
std::string tensor_on_same_gpu_if_not_optional_check(
    const std::string& var_names_str,
    const Tensors&... tensors) {
  std::optional<int64_t> gpu_index;
  bool on_same_gpu = true;

  // Collect the GPU index of the first non-empty optional tensor and make sure
  // that all tensors are on this same index.
  (
      [&](const auto& tensor) {
        if (!torch_tensor_undefined(tensor)) {
          return;
        }
        if (!torch_tensor_on_cuda_gpu_check(tensor)) {
          on_same_gpu = false;
          return;
        }
        const auto my_gpu_index = get_device_index_from_tensor(tensor);
        if (my_gpu_index) {
          if (!gpu_index) {
            gpu_index = my_gpu_index;
          } else if (*gpu_index != my_gpu_index) {
            on_same_gpu = false;
          }
        }
      }(tensors),
      ...);

  if (on_same_gpu) {
    return "";
  }

  std::vector<std::string> var_names;
  {
    std::string temp = "";
    for (const auto& x : var_names_str) {
      if (x == ',') {
        var_names.push_back(temp);
        temp = "";
      } else {
        temp.push_back(x);
      }
    }
    var_names.push_back(temp);
  }

  // Not all the tensors on a GPU or on the same GPU, generate a message.
  std::string msg = "Not all tensors were on the same GPU: ";
  size_t current_idx = 0;
  (
      [&](const auto& tensor) {
        if (current_idx > 0) {
          msg.append(", ");
        }
        msg.append(
            var_names.at(current_idx++) + "(" +
            torch_tensor_device_name(tensor));
        const auto gpu_device_index = get_device_index_from_tensor(tensor);
        if (gpu_device_index) {
          msg.append(":" + std::to_string(*gpu_device_index));
        }
        msg.append(")");
      }(tensors),
      ...);

  return msg;
}

// Generate constexpr array of variable names to improve diagnostic output and
// raise a message if any non-empty tensor is not on a GPU or not on the same
// GPU as all the other non-empty tensors.
#define TENSORS_ON_SAME_CUDA_GPU_IF_NOT_OPTIONAL(...)                        \
  do {                                                                       \
    const auto tensors_on_same_gpu =                                         \
        tensor_on_same_gpu_if_not_optional_check(#__VA_ARGS__, __VA_ARGS__); \
    TORCH_CHECK(tensors_on_same_gpu.empty(), tensors_on_same_gpu);           \
  } while (false)

/// Determine an appropriate CUDA block count along the x axis
///
/// When launching CUDA kernels the number of blocks B is often calculated
/// w.r.t. the number of threads T and items to be processed N as
/// B=(N+T-1)/T - which is integer division rounding up.
/// This function abstracts that calculation, performs it in an
/// overflow-safe manner, and limits the return value appropriately.
///
/// This is a general function for all integral data types.
/// The goal of this set of functions is to ensure correct calculations
/// across a variety of data types without forcing the programmer to
/// cast to an appropriate type (which is dangerous because we don't
/// have conversion warnings enabled). The values of the variables
/// can then be checked for correctness at run-time.
/// Specialized functions below handle various combinations of signed
/// and unsigned inputs. This system prevents "pointless comparison
/// against zero" warnings from the compiler for unsigned types
/// (simpler ways of suppressing this warning didn't work) while
/// maintaining the various warnings.
///
/// Function is designed to facilitate run-time value checking.
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<std::is_integral<Integer1>::value, bool> = true,
    std::enable_if_t<std::is_integral<Integer2>::value, bool> = true>
constexpr uint32_t cuda_calc_xblock_count_base(
    Integer1 num_items,
    Integer2 threads_per_block) {
  // The number of threads can be as high as 2048 on some newer architectures,
  // but this is not portable.
  TORCH_CHECK(threads_per_block <= 1024, "Number of threads must be <=1024!");
  // The CUDA specification at
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
  // states that for compute capability 3.5-* the grid dimension of a kernel
  // launch must must be <=2^31-1.
  constexpr uint64_t max_blocks = 2147483647;
  const auto u_num_items = static_cast<uint64_t>(num_items);
  const auto u_threads = static_cast<uint64_t>(threads_per_block);
  // Overflow safe variant of (a + b - 1) / b
  const uint64_t blocks =
      u_num_items / u_threads + (u_num_items % u_threads != 0);
  return static_cast<uint32_t>(std::min(blocks, max_blocks));
}

// See: cuda_calc_xblock_count_base
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<
        std::is_integral<Integer1>::value && std::is_signed<Integer2>::value,
        bool> = true,
    std::enable_if_t<
        std::is_integral<Integer2>::value && std::is_unsigned<Integer2>::value,
        bool> = true>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<
        std::is_integral<Integer1>::value && std::is_unsigned<Integer2>::value,
        bool> = true,
    std::enable_if_t<
        std::is_integral<Integer2>::value && std::is_signed<Integer2>::value,
        bool> = true>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  TORCH_CHECK(
      threads_per_block >= 0,
      "When calculating thread counts, the number of threads must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<
        std::is_integral<Integer1>::value && std::is_signed<Integer2>::value,
        bool> = true,
    std::enable_if_t<
        std::is_integral<Integer2>::value && std::is_signed<Integer2>::value,
        bool> = true>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  TORCH_CHECK(
      num_items >= 0,
      "When calculating block counts, the number of items must be positive!");
  TORCH_CHECK(
      threads_per_block >= 0,
      "When calculating thread counts, the number of threads must be positive!");
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

// See: cuda_calc_xblock_count_base
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<
        std::is_integral<Integer1>::value && std::is_unsigned<Integer2>::value,
        bool> = true,
    std::enable_if_t<
        std::is_integral<Integer2>::value && std::is_unsigned<Integer2>::value,
        bool> = true>
constexpr uint32_t cuda_calc_xblock_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  return cuda_calc_xblock_count_base(num_items, threads_per_block);
}

/// Determine an appropriate CUDA block count.
///
/// See cuda_calc_xblock_count_base() for details.
template <
    typename Integer1,
    typename Integer2,
    std::enable_if_t<std::is_integral<Integer1>::value, bool> = true,
    std::enable_if_t<std::is_integral<Integer2>::value, bool> = true>
constexpr uint32_t cuda_calc_block_count(
    Integer1 num_items,
    Integer2 threads_per_block) {
  // The CUDA specification at
  // https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications
  // states that the grid dimension of a kernel launch must generally
  // be <=65535. (For compute capability 3.5-* the grid's x-dimension must
  // be <=2^31-1.) Because this function does not know which dimension
  // is being calculated, we use the smaller limit.
  constexpr uint32_t max_blocks = 65535;
  return std::min(
      cuda_calc_xblock_count(num_items, threads_per_block), max_blocks);
}

// A wrapper class for passing dynamically sized dimension information (e.g.
// tensor.dims()) from the host to device.
constexpr size_t kStackArrayMaxDims = 5;

template <typename T>
struct StackArray {
  T vals[kStackArrayMaxDims];
  size_t ndim;
};

// Used in jagged_tensor_ops.cu and jagged_tensor_ops_cpu.cpp
// Passing lambda exp argument by value instead of by reference to avoid
// "internal compiler error: in maybe_undo_parenthesized_ref" error for specific
// compiler version.
#define JAGGED_TENSOR_DISPATCH_DIMS()                                         \
  AT_DISPATCH_INDEX_TYPES(x_offsets[0].scalar_type(), "jagged_indices", [=] { \
    switch (num_jagged_dim) {                                                 \
      case 1:                                                                 \
        INVOKE_KERNEL_WITH_DIM(1);                                            \
        break;                                                                \
      case 2:                                                                 \
        INVOKE_KERNEL_WITH_DIM(2);                                            \
        break;                                                                \
      case 3:                                                                 \
        INVOKE_KERNEL_WITH_DIM(3);                                            \
        break;                                                                \
      case 4:                                                                 \
        INVOKE_KERNEL_WITH_DIM(4);                                            \
        break;                                                                \
      case 5:                                                                 \
        INVOKE_KERNEL_WITH_DIM(5);                                            \
        break;                                                                \
      default:                                                                \
        TORCH_CHECK(                                                          \
            false, "unsupported number of jagged dim ", num_jagged_dim);      \
    }                                                                         \
  });

// TODO: Merge this with the device code
template <typename scalar_t>
void binary_search_range_cpu(
    int* found,
    const scalar_t* arr,
    const scalar_t target,
    const int num_entries) {
  const int last_entry = num_entries - 1;
  int start = 0, end = last_entry;
  int found_ = -1;
  while (start <= end) {
    int mid = start + (end - start) / 2;
    scalar_t mid_offset = arr[mid];
    if (target == mid_offset) {
      if (mid != last_entry && target != arr[last_entry]) {
        // Do linear scan in case of duplicate data (We assume that the
        // number of duplicates is small.  This can we very bad if the
        // number of duplicates is large)
        for (int i = mid + 1; i < num_entries; i++) {
          if (target != arr[i]) {
            found_ = i;
            break;
          }
        }
      }
      break;
    } else if (target < mid_offset) {
      if (mid == 0) {
        found_ = 0;
        break;
      } else if (mid - 1 >= 0 && target > arr[mid - 1]) {
        found_ = mid;
        break;
      }
      end = mid - 1;
    } else {
      if (mid + 1 <= last_entry && target < arr[mid + 1]) {
        found_ = mid + 1;
        break;
      }
      start = mid + 1;
    }
  }
  *found = found_;
}

template <int x>
struct log2_calc_ {
  enum { value = log2_calc_<(x >> 1)>::value + 1 };
};
template <>
struct log2_calc_<0> {
  enum { value = 0 };
};

template <int x>
struct log2_calc {
  enum { value = log2_calc_<(x - 1)>::value };
};
#if 0
template <>
struct log2_calc<0> { enum { value = 0 }; };
template <>
struct log2_calc<1> { enum { value = 0 }; };
#endif
