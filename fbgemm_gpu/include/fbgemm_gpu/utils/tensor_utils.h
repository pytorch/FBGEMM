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

inline bool torch_tensor_on_cpu_check(const at::Tensor& ten) {
  return ten.is_cpu();
}

inline bool torch_tensor_on_cpu_check(const std::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_cpu_check(ten.value());
}

inline std::optional<int64_t> get_device_index_from_tensor(
    const at::Tensor& ten) {
  return {ten.device().index()};
}

inline std::optional<int64_t> get_device_index_from_tensor(
    const std::optional<at::Tensor>& ten) {
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
    const std::optional<at::Tensor>& ten) {
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
    const std::optional<at::Tensor>& ten2) {
  return !ten2.has_value() || ten1.get_device() == ten2->get_device();
}

inline bool torch_tensor_undefined(const at::Tensor& ten) {
  return ten.defined();
}

inline bool torch_tensor_undefined(const std::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_undefined(ten.value());
}

inline bool torch_tensor_on_cuda_gpu_check(const at::Tensor& ten) {
  return ten.is_cuda();
}

inline bool torch_tensor_on_cuda_gpu_check(
    const std::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_on_cuda_gpu_check(ten.value());
}

inline bool torch_tensor_empty_or_on_cuda_gpu_check(const at::Tensor& ten) {
  return (ten.numel() == 0) || ten.is_cuda();
}

inline bool torch_tensor_empty_or_on_cuda_gpu_check(
    const std::optional<at::Tensor>& ten) {
  return !ten.has_value() ||
      torch_tensor_empty_or_on_cuda_gpu_check(ten.value());
}

inline bool torch_tensor_empty_or_on_cpu_check(const at::Tensor& ten) {
  return (ten.numel() == 0) || ten.is_cpu();
}

inline bool torch_tensor_empty_or_on_cpu_check(
    const std::optional<at::Tensor>& ten) {
  return !ten.has_value() || torch_tensor_empty_or_on_cpu_check(ten.value());
}

inline bool torch_tensor_on_cpu_or_on_mtia_check(const at::Tensor& ten) {
  return ten.is_cpu() || ten.is_mtia();
}

#define TENSOR_ON_CPU_OR_MTIA(x)                                      \
  TORCH_CHECK(                                                        \
      torch_tensor_on_cpu_or_on_mtia_check(x),                        \
      #x " must be a CPU or MTIA tensor; it is currently on device ", \
      torch_tensor_device_name(x))

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

#define TENSOR_NUMEL_IS_GT(ten, num)                \
  TORCH_CHECK(                                      \
      (ten).numel() > (num),                        \
      "Tensor '" #ten "' must have more than " #num \
      " element(s). "                               \
      "Found ",                                     \
      (ten).numel())

#define TENSORS_HAVE_SAME_SYM_NUMEL(x, y)                              \
  TORCH_SYM_CHECK(                                                     \
      x.sym_numel().sym_eq(y.sym_numel()),                             \
      #x " must have the same number of elements as " #y " They had ", \
      (x).sym_numel(),                                                 \
      " and ",                                                         \
      (y).sym_numel())

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

inline at::Tensor aligned_grad_output_tensor_for_cuda_backwards(
    const at::Tensor& grad_output) {
  auto aligned_grad_output = grad_output;
  // FIXME: to support aligned memory access in Vec4T load/store function
  // 16 for FP32 and 8 for FP16
  if (grad_output.dim() > 1 &&
      (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0 ||
       grad_output.stride(1) != 1 || grad_output.stride(0) % 4 != 0)) {
    aligned_grad_output = grad_output.contiguous();
  }
  if (reinterpret_cast<uint64_t>(grad_output.data_ptr()) % 16 != 0) {
    aligned_grad_output = at::empty_like(grad_output).copy_(grad_output);
  }
  return aligned_grad_output;
}

template <typename... ScalarTypes>
std::string tensor_scalar_type_is_one_of(
    const at::Tensor& ten,
    const ScalarTypes&... ttypes) {
  auto has_match = false;

  (
      [&](const auto& ttype) {
        if (ten.scalar_type() == ttype) {
          has_match = true;
        }
      }(ttypes),
      ...);

  if (has_match) {
    return "";
  }

  std::string msg = "Tensor's scalar type (";
  msg.append(toString(ten.scalar_type()));
  msg.append(") did not match any one of the following types: [");
  (
      [&](const auto& ttype) {
        msg.append(toString(ttype));
        msg.append(", ");
      }(ttypes),
      ...);

  msg.append("]");
  return msg;
}

#define TENSOR_SCALAR_TYPE_IS_ONE_OF(...)                             \
  do {                                                                \
    const auto has_match = tensor_scalar_type_is_one_of(__VA_ARGS__); \
    TORCH_CHECK(has_match.empty(), has_match);                        \
  } while (false)

template <typename... Tensors>
std::string tensors_have_same_scalar_type(const Tensors&... tensors) {
  std::optional<at::ScalarType> dtype;
  bool have_same_type = true;

  (
      [&](const auto& tensor) {
        if (!dtype) {
          dtype = tensor.scalar_type();
        } else if (*dtype != tensor.scalar_type()) {
          have_same_type = false;
        }
      }(tensors),
      ...);

  if (have_same_type) {
    return "";
  }

  std::string msg = "Tensors' scalar types (";
  (
      [&](const auto& tensor) {
        msg.append(toString(tensor.scalar_type()));
        msg.append(", ");
      }(tensors),
      ...);
  msg.append(") are not one and the same!");
  return msg;
}

#define TENSORS_HAVE_SAME_SCALAR_TYPE(...)                                  \
  do {                                                                      \
    const auto have_same_type = tensors_have_same_scalar_type(__VA_ARGS__); \
    TORCH_CHECK(have_same_type.empty(), have_same_type);                    \
  } while (false)
