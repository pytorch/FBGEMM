/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * All rights reserved.
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

// clang-format off
#pragma once

namespace fbgemm_gpu {

/* This file is code-generated from generate_backward_split.py to enumerate index of
 * auxiliary arguments in the list. The enum is generated from the dict `aux_args`,
 * which is a single source that controls and maintains the argument position.
*/

{%- for name in aux_names %}
enum ArgIndex_{{ name }} {
  {%- for var in aux_args[name] %}
  IDX_{{ var | upper }} = {{ loop.index - 1 }},
  {%- endfor %}
  {{ name | upper }}_SIZE = {{ name | length }}
};
{%- endfor %}

namespace utils {
  template<typename T>
  auto list_get(const c10::List<T>& list, const int32_t idx, const T& default_value) {      
    static_assert(
      std::is_same_v<
          T,
          std::remove_cv_t<
              std::remove_pointer_t<std::remove_reference_t<T>>>>,
      "T must be a pure type (no pointers, references, or cv-qualifiers)");

    if (idx < 0 || idx >= list.size()) {
      return default_value;
    }

    return list[idx];
  }
}


} // namespace fbgemm_gpu
