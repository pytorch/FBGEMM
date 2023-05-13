/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <string>
#include <tuple>
#include <vector>

namespace fbgemm_gpu {

#define FBGEMM_GPU_ENUM_CREATE_TAG(module_name)                         \
  struct fbgemm_gpu_enum_tag_##module_name {};                          \
  template <>                                                           \
  enum_registration<struct fbgemm_gpu_enum_tag_##module_name>*          \
      enum_registration<                                                \
          struct fbgemm_gpu_enum_tag_##module_name>::registration_list; \
  extern template class enum_registration<                              \
      struct fbgemm_gpu_enum_tag_##module_name>;

#define FBGEMM_GPU_ENUM_TAG(module_name) \
  struct fbgemm_gpu_enum_tag_##module_name

#define FBGEMM_GPU_ENUM_GLOGAL(module_name)                                    \
  template class enum_registration<FBGEMM_GPU_ENUM_TAG(module_name)>;          \
  template <>                                                                  \
  enum_registration<FBGEMM_GPU_ENUM_TAG(module_name)>*                         \
      enum_registration<FBGEMM_GPU_ENUM_TAG(module_name)>::registration_list = \
          nullptr;

// To work around (escape from) hipify_torch, the names of the idendifiers
// are decoposed to `prefix` and `enum_name`.
#define FBGEMM_GPU_ENUM_REGISTER_START(module_name, prefix, enum_name)     \
  enum_registration<FBGEMM_GPU_ENUM_TAG(module_name)> fbgemm_fpu_enum_reg_ \
      ## prefix ## enum_name( #prefix #enum_name,

#define FBGEMM_GPU_ENUM_REGISTER_END );

#define FBGEMM_GPU_ENUM_OP(module_name, op_name) \
#op_name "() -> ((str, (str, int)[])[])",      \
      TORCH_FN(enum_query <FBGEMM_GPU_ENUM_TAG(module_name)>)
// To work around (escape from) hipify_torch, the names of the idendifiers
// are decoposed to `x` and `y`. `z` is supposed to be hipified.
#define FBGEMM_GPU_ENUM_ITEM(x, y, z) \
  { #x #y, z }

using enum_item = std::tuple<std::string, int64_t>;

using enum_items = std::vector<enum_item>;

using enum_result = std::vector<
    std::tuple<std::string, std::vector<std::tuple<std::string, int64_t>>>>;

template <class T>
class enum_registration {
 public:
  enum_registration(const char* enum_name, enum_items&& items)
      : name_(enum_name), items_(std::move(items)) {
    next_ = registration_list;
    registration_list = this;
  }

  static enum_result enum_query() {
    enum_result result;

    for (auto next = registration_list; next != nullptr; next = next->next_) {
      result.emplace_back(
          std::make_tuple(std::string(next->name_), next->items_));
    }

    return result;
  }

 protected:
  static enum_registration<T>* registration_list;

  enum_registration<T>* next_;
  const char* name_;
  std::vector<enum_item> items_;
};

template <class T>
static inline enum_result enum_query() {
  return enum_registration<T>::enum_query();
}

} // namespace fbgemm_gpu
